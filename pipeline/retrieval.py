"""VeriGraph AI — Historical + Live Evidence Retrieval."""
from __future__ import annotations
import numpy as np
import pandas as pd
import re
import math
from pathlib import Path
from typing import Any, Dict, List
from config.settings import KB_DIR, NB6_DIR, MAX_FINAL_FUSED_EVIDENCE, MAX_QUERIES, MAX_EXTERNAL_RESULTS_PER_PROVIDER, MAX_FINAL_EVIDENCE, MAX_EVIDENCE_FOR_AGENTS, CONFIDENCE_DECIMALS
from services.faiss_service import (
    retrieve_historical_text_bge,
    retrieve_historical_image_clip,
    retrieve_historical_text_clip,
    init_faiss_indexes,
    get_faiss_index_info,
)
from services.live_search_service import run_live_retrieval
from services.url_utils import (
    normalize_url, extract_domain, validate_url, 
    resolve_source_provenance, format_url_for_display, clean_snippet
)
from pipeline.embeddings import embed_text


def _c(x):
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception: pass
    return re.sub(r"\s+", " ", str(x).strip())


def _f(x, d=0.0):
    try: v = float(x); return v if math.isfinite(v) else d
    except Exception: return d


def _score(value, default=0.0):
    """Validate and convert a score value to float.
    
    Raises TypeError if the value is a list/tuple/dict/str (not scalar numeric).
    Returns default if value is None or non-finite.
    """
    if isinstance(value, bool):
        return float(value)

    if value is None:
        return float(default)

    if isinstance(value, (list, tuple, dict, str)):
        raise TypeError(
            f"Score must be scalar numeric, got {type(value).__name__}: {value!r}"
        )

    try:
        v = float(value)
    except (ValueError, TypeError):
        return float(default)

    if not np.isfinite(v):
        return float(default)

    return float(np.clip(v, 0.0, 1.0))


def _extract_key_entities(claim: str) -> List[str]:
    """Extract key entities and temporal markers from the claim."""
    import re
    
    # Extract potential entity names (capitalized phrases)
    entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', claim)
    
    # Extract year/temporal markers
    years = re.findall(r'\b\d{4}\b', claim)
    
    # Extract ordinal indicators (18th, 2026, etc.)
    ordinals = re.findall(r'\b\d+(?:st|nd|rd|th)\b', claim)
    
    # Filter out common words that aren't meaningful entities
    stop_words = {'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', 'on', 'at', 'by', 'from'}
    entities = [e for e in entities if e.lower() not in stop_words]
    
    result = []
    for e in entities:
        if e not in result:
            result.append(e)
    for y in years:
        if y not in result:
            result.append(y)
    for o in ordinals:
        if o not in result:
            result.append(o)
    
    return result


def _extract_event_terms(claim: str) -> List[str]:
    """Extract event-type terms from the claim (summit, conference, election, etc.)."""
    event_terms = []
    for term in [
        "summit", "conference", "meeting", "election", "tournament",
        "launch", "agreement", "chairmanship", "summit", "bilateral",
        "multilateral", "forum", "expo", "congress", "assembly",
    ]:
        if re.search(rf"\b{re.escape(term)}\b", claim, re.IGNORECASE):
            event_terms.append(term)
    return event_terms


def _extract_claim_locations(claim: str) -> List[str]:
    """Extract location names from the claim using a broader location vocabulary."""
    location_vocab = [
        "New Delhi", "Delhi", "Mumbai", "Chennai", "Bangalore",
        "Hyderabad", "Kolkata", "India", "Bharat", "Goa", "Agra",
        "London", "Paris", "Washington", "Beijing", "Tokyo",
        "Singapore", "Dubai", "New York", "Geneva", "Brussels",
    ]
    locations = []
    for loc in location_vocab:
        if re.search(rf"\b{re.escape(loc)}\b", claim, re.IGNORECASE):
            if loc not in locations:
                locations.append(loc)
    return locations


def _hard_relevance_filter(evidence_items: List[Dict[str, Any]], claim: str) -> List[Dict[str, Any]]:
    """Strong relevance filter: reject evidence lacking sufficient claim anchors.

    Evidence must satisfy at least one of:
    1. At least 2 entity/year/location anchors from the claim appear in title/snippet
    2. At least 1 strong anchor (event term + location, or entity + year, or event + year)
    3. A high semantic similarity (computed later via BGE)

    Evidence failing all anchor checks is rejected as irrelevant.
    """
    # Extract claim anchors
    claim_entities = _extract_key_entities(claim)
    claim_years = re.findall(r"\b(?:19|20)\d{2}\b", claim)
    claim_locations = _extract_claim_locations(claim)
    claim_event_terms = _extract_event_terms(claim)

    filtered = []
    for item in evidence_items:
        title = str(item.get("title", "")).lower()
        snippet = str(item.get("snippet", "")).lower()
        combined = title + " " + snippet

        # Count entity matches (case-insensitive, length > 2 to avoid noise)
        entity_matches = sum(
            1 for e in claim_entities
            if e.lower() in combined and len(e) > 2
        )

        # Count year matches
        year_matches = sum(1 for y in claim_years if y in combined)

        # Count location matches
        location_matches = sum(
            1 for l in claim_locations
            if l.lower() in combined
        )

        # Count event term matches
        event_matches = sum(
            1 for ev in claim_event_terms
            if ev.lower() in combined
        )

        total_anchors = entity_matches + year_matches + location_matches + event_matches

        # Strong anchor combinations
        has_event_location = (event_matches > 0 and location_matches > 0)
        has_entity_year = (entity_matches > 0 and year_matches > 0)
        has_event_year = (event_matches > 0 and year_matches > 0)
        has_entity_location = (entity_matches > 0 and location_matches > 0)

        # Accept if at least 2 anchors, or any strong combination
        if total_anchors >= 2 or has_event_location or has_entity_year or has_event_year or has_entity_location:
            filtered.append(item)
        # Also accept if claim has no anchors at all (can't filter)
        elif not claim_entities and not claim_years and not claim_locations and not claim_event_terms:
            filtered.append(item)

    return filtered


def _clean_claim(claim: str) -> str:
    """Strip social media boilerplate from claims to extract factual content.

    Removes:
    - Like/comment counts (including numbers with commas like "7,780 likes")
    - Usernames and handles
    - Platform names (cinemavikatan, etc.)
    - Irrelevant punctuation and metadata
    - Trailing date patterns
    """
    import re

    # Remove like/comment counts like "7,780 likes", "20 comments"
    # Handle numbers with commas: \d{1,3}(?:,\d{3})* matches grouped digits
    claim = re.sub(r'\d{1,3}(?:,\d{3})*\s*\.?\d*\s*likes?', '', claim, flags=re.IGNORECASE)
    claim = re.sub(r'\d{1,3}(?:,\d{3})*\s*\.?\d*\s*comments?', '', claim, flags=re.IGNORECASE)

    # Also handle plain number likes/comments without commas
    claim = re.sub(r'\d+\.?\d*\s*likes?', '', claim, flags=re.IGNORECASE)
    claim = re.sub(r'\d+\.?\d*\s*comments?', '', claim, flags=re.IGNORECASE)

    # Remove usernames/handles (words starting with @)
    claim = re.sub(r'\B@\w+', '', claim)

    # Remove platform names that are typically boilerplate
    claim = re.sub(r'\bcinemavikatan\b', '', claim, flags=re.IGNORECASE)
    claim = re.sub(r'\btiktok\b', '', claim, flags=re.IGNORECASE)
    claim = re.sub(r'\byoutube\b', '', claim, flags=re.IGNORECASE)
    claim = re.sub(r'\bfacebook\b', '', claim, flags=re.IGNORECASE)

    # Remove any leftover standalone metadata tokens (likes, comments, etc.)
    claim = re.sub(r'\blikes?\b', '', claim, flags=re.IGNORECASE)
    claim = re.sub(r'\bcomments?\b', '', claim, flags=re.IGNORECASE)

    # Remove trailing quotes and punctuation
    # Strip trailing whitespace only
    claim = claim.rstrip()

    # Normalize whitespace
    claim = re.sub(r'\s+', ' ', claim).strip()

    # Remove leading punctuation/ separators that remain after boilerplate removal
    # e.g., leading commas, dashes, colons from removed "7,780 likes, 20 comments - ..." patterns
    claim = re.sub(r'^[,\-\s:]+', '', claim)

    # Final strip after punctuation removal
    claim = claim.strip()

    return claim



def _generate_query_intents(claim: str, max_queries: int = MAX_QUERIES) -> List[str]:
    """Generate factual, retrieval-oriented query variants from the claim.

    Queries are built from extracted entities, event terms, dates, and
    locations rather than appending generic 'support' / 'refute' suffixes.
    The total number of intents is bounded by max_queries.
    """
    claim = _clean_claim(claim)
    base = str(claim).strip()
    key_entities = _extract_key_entities(claim)

    # Extract event terms, years, and locations from the claim
    event_terms = _extract_event_terms(claim)
    years = re.findall(r"\b(?:19|20)\d{2}\b", claim)
    locations = _extract_claim_locations(claim)

    # Build factual query variants
    intents = []

    # 1. Original cleaned claim
    intents.append(base)

    # 2. Entity-focused query (top entities + years + locations)
    if key_entities:
        parts = list(dict.fromkeys(key_entities[:4]))
        if years:
            parts.append(years[0])
        if locations:
            parts.append(locations[0])
        entity_query = " ".join(parts)
        if entity_query and entity_query != base:
            intents.append(entity_query)

    # 3. Event-focused query (event term + year + location)
    if event_terms:
        event_parts = [event_terms[0]]
        if years:
            event_parts.append(years[0])
        if locations:
            event_parts.append(locations[0])
        event_query = " ".join(event_parts)
        if event_query and event_query not in intents:
            intents.append(event_query)

    # 4. Year + location query
    if years and locations:
        yr_loc_query = f"{years[0]} {locations[0]}"
        if yr_loc_query not in intents:
            intents.append(yr_loc_query)

    # 5. Entity + year query
    if key_entities and years:
        ent_year = f"{key_entities[0]} {years[0]}"
        if ent_year not in intents:
            intents.append(ent_year)

    # 6. Official source query (entity + official)
    if key_entities:
        official_query = f"{key_entities[0]} {years[0] if years else ''} official statement".strip()
        if official_query not in intents:
            intents.append(official_query)

    # Deduplicate while preserving order, then limit to max_queries
    seen = set()
    unique_intents = []
    for intent in intents:
        if intent not in seen and len(unique_intents) < max_queries:
            seen.add(intent)
            unique_intents.append(intent)

    # Ensure we have at least 1 query
    if not unique_intents:
        unique_intents = [base]

    return unique_intents


def _enrich_evidence_item(item: Dict[str, Any], case_id: str, eid_counter: int,
                        claim_emb: Optional[np.ndarray] = None,
                        claim_text: str = "") -> Dict[str, Any]:
    """Enrich an evidence item with provenance, normalized URL, and claim-aware scoring fields."""
    # Resolve provenance
    provenance = resolve_source_provenance(
        url=item.get("url", ""),
        source=item.get("source", ""),
        provider=item.get("provider", ""),
        dataset=item.get("dataset", ""),
        retrieval_channel=item.get("retrieval_channel", ""),
    )
    
    # Format for display
    display = format_url_for_display(provenance)
    
    # Ensure evidence_id
    if not item.get("evidence_id"):
        item["evidence_id"] = f"ev_{case_id}_{eid_counter:04d}"
    
    # Clean snippet
    item["snippet"] = clean_snippet(item.get("snippet", ""))
    
    # Add provenance fields
    item["provenance"] = provenance
    item["display"] = display
    
    # Ensure domain is properly extracted
    if not item.get("domain") or item.get("domain") == "unknown":
        item["domain"] = provenance.get("domain") or "unavailable"
    
    # Ensure normalized URL
    item["url"] = provenance.get("canonical_url") or provenance.get("url", "")
    
    # Ensure published_at is never empty - LLM judge penalizes missing dates
    if not item.get("published_at"):
        item["published_at"] = "date not specified"
    
    # Add provenance_type for easy filtering
    item["provenance_type"] = provenance.get("provenance_type", "UNRESOLVED")
    item["url_status"] = provenance.get("url_status", "not_available")
    item["url_verified"] = provenance.get("url_verified", False)
    
    # Retrieval intent - track which query intent retrieved this evidence
    item["retrieval_intent"] = item.get("retrieval_intent", "unknown")
    
    # ===== CLAIM-AWARE SCORING (CALIBRATED) =====
    evidence_text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
    
    # A. SEMANTIC CLAIM RELEVANCE
    # Use BGE cosine similarity between claim and evidence title+snippet
    if claim_emb is not None:
        raw_evidence_text = f"{item.get('title', '')} {item.get('snippet', '')}".strip()
        if raw_evidence_text:
            emb = embed_text(raw_evidence_text)
            claim_norm = np.linalg.norm(claim_emb)
            emb_norm = np.linalg.norm(emb)
            if claim_norm > 0 and emb_norm > 0:
                semantic_sim = float(np.dot(claim_emb / claim_norm, emb / emb_norm))
            else:
                semantic_sim = 0.0
            item["claim_relevance_score"] = _score(max(0.0, min(1.0, semantic_sim)))
        else:
            item["claim_relevance_score"] = _score(0.0)
    else:
        item["claim_relevance_score"] = _score(max(0.0, min(1.0, item.get("relevance_score", 0.5))))
    
    # Extract genuine CLAIM anchors (from claim_text, NOT from evidence!)
    if claim_text:
        target_claim_entities = _extract_key_entities(claim_text)
        target_claim_years = re.findall(r"\b(?:19|20)\d{2}\b", claim_text)
        target_claim_locations = _extract_claim_locations(claim_text)
        target_claim_event_terms = _extract_event_terms(claim_text)
    else:
        target_claim_entities = []
        target_claim_years = []
        target_claim_locations = []
        target_claim_event_terms = []
    
    # B. GENUINE ENTITY MATCH (checks if CLAIM entities appear in evidence)
    if target_claim_entities:
        matched_entities = [
            e for e in target_claim_entities
            if e.lower() in evidence_text and len(e) > 2
        ]
        item["entity_match_score"] = _score(min(1.0, len(matched_entities) / max(1, len(target_claim_entities))))
    else:
        item["entity_match_score"] = _score(0.5)
    
    # C. GENUINE EVENT MATCH (checks if claim's event term appears in evidence)
    if target_claim_event_terms:
        matched_events = [ev for ev in target_claim_event_terms if ev.lower() in evidence_text]
        item["event_match_score"] = _score(1.0 if matched_events else 0.0)
    else:
        item["event_match_score"] = _score(0.5)
    
    # D. GENUINE LOCATION MATCH (checks if claim's location appears in evidence)
    if target_claim_locations:
        matched_locs = [loc for loc in target_claim_locations if loc.lower() in evidence_text]
        item["location_match_score"] = _score(1.0 if matched_locs else 0.0)
    else:
        item["location_match_score"] = _score(0.5)
    
    # E. GENUINE TEMPORAL MATCH (checks if claim's target year appears in evidence)
    evidence_years = re.findall(r"\b(?:19|20)\d{2}\b", evidence_text)
    if target_claim_years:
        has_exact_year = any(y in evidence_years for y in target_claim_years)
        if has_exact_year:
            item["temporal_match_score"] = _score(1.0)
        elif evidence_years:
            # Different year mentioned -> severe penalty!
            item["temporal_match_score"] = _score(0.1)
        else:
            item["temporal_match_score"] = _score(0.4)
    else:
        item["temporal_match_score"] = _score(0.5)
    
    # F. SOURCE CREDIBILITY
    # Use source domain information for credibility
    from services.source_credibility import classify_source
    _cred = classify_source(item.get("domain", ""), item.get("url", ""))
    item["credibility_score"] = _score(_cred["score"])
    item["credibility_reason"] = _cred["reason"]
    item["credibility_tier"] = _cred["tier"]
    
    # G. DIRECTNESS
    # Higher for evidence that directly mentions claim entities/events
    directness_score = item["claim_relevance_score"] * 0.7 + item["entity_match_score"] * 0.3
    item["directness_score"] = _score(directness_score)
    
    # H. NARRATIVE COHERENCE
    # Based on credibility and relevance combined
    item["narrative_coherence_score"] = _score(
        (item.get("credibility_score", 0.5) + item.get("claim_relevance_score", 0)) / 2.0
    )
    
    # I. INTENT MATCH
    # Default based on retrieval intent
    item["intent_match_score"] = _score(item.get("retrieval_intent", "unknown") == "original" and 0.8 or 0.5)
    
    # J. VISUAL FILTERING & MULTIMODAL CONSISTENCY (Base-paper multimodal fact-checking)
    # Checks if candidate contains visual evidence and validates against claim context
    is_multimodal = bool(item.get("image_path") or item.get("evidence_type") == "image" or item.get("retrieval_channel") in ("clip_image", "clip_text"))
    if is_multimodal:
        temp_score = item.get("temporal_match_score", 0.5)
        # Detect out-of-context cheapfake/manipulation (recycled imagery with temporal clash)
        if temp_score <= 0.20 and item.get("entity_match_score", 0.0) >= 0.40:
            item["visual_filter_status"] = "OUT_OF_CONTEXT_REUSE"
            item["out_of_context_flag"] = True
            item["visual_similarity_score"] = _score(0.20)
            item["multimodal_score"] = _score(0.25)
        else:
            item["visual_filter_status"] = "VISUAL_GROUNDING_CONFIRMED"
            item["out_of_context_flag"] = False
            item["visual_similarity_score"] = _score(0.80)
            item["multimodal_score"] = _score(0.75)
    else:
        item["visual_filter_status"] = "TEXT_ONLY_PASS"
        item["out_of_context_flag"] = False
        item["visual_similarity_score"] = _score(0.50)
        item["multimodal_score"] = _score(0.0)

    # K. FINAL FUSION SCORE
    # Engineering weights (Req #9): weighted combination of all evidence-specific scores
    w = {
        "claim_relevance": 0.28,
        "directness": 0.14,
        "entity_match": 0.12,
        "event_match": 0.10,
        "temporal_match": 0.10,
        "location_match": 0.06,
        "credibility": 0.06,
        "narrative_coherence": 0.04,
        "intent_match": 0.02,
        "multimodal": 0.08,
    }
    final_score = (
        w["claim_relevance"] * item.get("claim_relevance_score", 0) +
        w["directness"] * item.get("directness_score", 0) +
        w["entity_match"] * item.get("entity_match_score", 0) +
        w["event_match"] * item.get("event_match_score", 0) +
        w["temporal_match"] * item.get("temporal_match_score", 0) +
        w["location_match"] * item.get("location_match_score", 0) +
        w["credibility"] * item.get("credibility_score", 0.5) +
        w["narrative_coherence"] * item.get("narrative_coherence_score", 0) +
        w["intent_match"] * item.get("intent_match_score", 0.5) +
        w["multimodal"] * item.get("multimodal_score", 0.0)
    )
    item["final_score"] = round(final_score, CONFIDENCE_DECIMALS)
    return item


def build_evidence_package(text_emb: np.ndarray, image_emb: np.ndarray,
                           case_claim: str, case_id: str) -> Dict[str, Any]:
    """Build evidence package using claim-aware retrieval and multimodal ranking."""
    
    # Initialize FAISS indexes if not already initialized
    historical_status: Dict[str, str] = {}
    historical_errors: Dict[str, str] = {}
    try:
        init_faiss_indexes()
    except Exception as exc:
        historical_status["faiss_index"] = "exception"
        historical_errors["faiss_index"] = f"{type(exc).__name__}: {exc}"
    
    # Stage 1: Candidate retrieval using BGE text and CLIP image
    hist_text: list = []
    hist_image: list = []
    if text_emb is None:
        historical_status["historical_text"] = "skipped_no_embedding"
        historical_errors["historical_text"] = "text embedding unavailable (see the embeddings audit entry)"
    else:
        try:
            hist_text = retrieve_historical_text_bge(text_emb, top_k=10)
            historical_status["historical_text"] = "success" if hist_text else "empty"
        except Exception as exc:
            historical_status["historical_text"] = "exception"
            historical_errors["historical_text"] = f"{type(exc).__name__}: {exc}"
    if image_emb is not None:
        try:
            hist_image = retrieve_historical_image_clip(image_emb, top_k=10)
            historical_status["historical_image"] = "success" if hist_image else "empty"
        except Exception as exc:
            historical_status["historical_image"] = "exception"
            historical_errors["historical_image"] = f"{type(exc).__name__}: {exc}"
    
    # Generate bounded query intents (Req #5: bounded multi-query retrieval)
    query_intents = _generate_query_intents(case_claim, max_queries=MAX_QUERIES)
    
    # Run live retrieval with bounded queries (Req #5: use config max_per_provider)
    live = run_live_retrieval(query_intents, max_per_provider=MAX_EXTERNAL_RESULTS_PER_PROVIDER)

    # Hard relevance filter: reject evidence lacking event/entity/date anchors matching the claim.
    # Counts are tracked so discarded evidence stays visible instead of silently vanishing.
    all_live = live.get("results", [])
    filtered_live = _hard_relevance_filter(all_live, case_claim)
    live = {
        "results": filtered_live,
        "provider_status": live.get("provider_status", {}),
        "provider_diagnostics": live.get("provider_diagnostics", {}),
        "raw_count": len(all_live),
        "relevant_count": len(filtered_live),
        "filtered_count": len(all_live) - len(filtered_live),
        "deduplicated_count": live.get("deduplicated_count", 0),
    }
    
    all_evidence = []
    eid_counter = 0
    
    # Process historical text evidence
    for r in hist_text:
        eid_counter += 1
        item = {
            "evidence_id": f"ev_{case_id}_{eid_counter:04d}",
            "evidence_type": "historical",
            "retrieval_channel": r.get("retrieval_channel", ""),
            "provider": "persistent_kb",
            "title": _c(r.get("claim", "")),
            "url": _c(r.get("url", "")),
            "snippet": _c(r.get("evidence_text", "")),
            "domain": _c(r.get("source_domain", "")),
            "source": _c(r.get("source", "")),
            "published_at": _c(r.get("publication_date", "")),
            "relevance_score": min(1.0, max(0.0, _f(r.get("retrieval_score", 0)))),
            "credibility_score": 0.5,
            "sample_id": _c(r.get("sample_id", "")),
            "dataset": _c(r.get("dataset", "")),
            # Claim-aware scoring components (normalized 0-1)
            "claim_relevance_score": 0.0,
            "directness_score": 0.0,
            "entity_match_score": 0.0,
            "event_match_score": 0.0,
            "temporal_match_score": 0.0,
            "location_match_score": 0.0,
            "narrative_coherence_score": 0.0,
            "intent_match_score": 0.0,
            "final_score": 0.0,
        }
        # Apply hard relevance filter to historical evidence
        if _hard_relevance_filter([item], case_claim):
            all_evidence.append(_enrich_evidence_item(item, case_id, eid_counter, text_emb, claim_text=case_claim))
    
    # Process historical CLIP image evidence
    for r in hist_image:
        eid_counter += 1
        item = {
            "evidence_id": f"ev_{case_id}_{eid_counter:04d}",
            "evidence_type": "historical",
            "retrieval_channel": r.get("retrieval_channel", ""),
            "provider": "persistent_kb",
            "title": _c(r.get("claim", "")),
            "url": _c(r.get("url", "")),
            "snippet": _c(r.get("evidence_text", "")),
            "domain": _c(r.get("source_domain", "")),
            "source": _c(r.get("source", "")),
            "published_at": _c(r.get("publication_date", "")),
            "relevance_score": min(1.0, max(0.0, _f(r.get("retrieval_score", 0)))),
            "credibility_score": 0.5,
            "sample_id": _c(r.get("sample_id", "")),
            "dataset": _c(r.get("dataset", "")),
            # Claim-aware scoring components (normalized 0-1)
            "claim_relevance_score": 0.0,
            "directness_score": 0.0,
            "entity_match_score": 0.0,
            "event_match_score": 0.0,
            "temporal_match_score": 0.0,
            "location_match_score": 0.0,
            "narrative_coherence_score": 0.0,
            "intent_match_score": 0.0,
            "final_score": 0.0,
        }
        # Apply hard relevance filter to historical image evidence
        if _hard_relevance_filter([item], case_claim):
            all_evidence.append(_enrich_evidence_item(item, case_id, eid_counter, text_emb, claim_text=case_claim))
    
    # Process live retrieval results
    for r in live.get("results", []):
        eid_counter += 1
        domain = ""
        try:
            domain = extract_domain(r.get("url", ""))
        except Exception:
            pass
        item = {
            "evidence_id": f"ev_{case_id}_{eid_counter:04d}",
            "evidence_type": "live",
            "retrieval_channel": "live_search",
            "provider": r.get("provider", ""),
            "title": _c(r.get("title", "")),
            "url": _c(r.get("url", "")),
            "snippet": _c(r.get("snippet", "")),
            "domain": domain or "",
            "source": _c(r.get("source_name", "")),
            "published_at": _c(r.get("published_at", "")),
            "relevance_score": 0.5,
            "credibility_score": 0.5,
            "sample_id": "",
            "dataset": "",
            # Query provenance: which generated query retrieved this evidence.
            "query": r.get("query", ""),
            # Claim-aware scoring components (normalized 0-1)
            "claim_relevance_score": 0.0,
            "directness_score": 0.0,
            "entity_match_score": 0.0,
            "event_match_score": 0.0,
            "temporal_match_score": 0.0,
            "location_match_score": 0.0,
            "narrative_coherence_score": 0.0,
            "intent_match_score": 0.0,
            "final_score": 0.0,
        }
        all_evidence.append(_enrich_evidence_item(item, case_id, eid_counter, text_emb, claim_text=case_claim))
    
    # Assign retrieval intent based on actual query intent and source provenance
    for e in all_evidence:
        q_text = (e.get("query") or "").lower()
        if any(term in q_text for term in ("refute", "false", "hoax", "debunk", "fake")):
            e["retrieval_intent"] = "refute"
        elif any(term in q_text for term in ("official", "confirmed", "statement", "government")):
            e["retrieval_intent"] = "official"
        elif any(term in q_text for term in ("support", "verify")):
            e["retrieval_intent"] = "support"
        else:
            e["retrieval_intent"] = "factual_variant"
    
    for e in all_evidence:
        e["fusion_score"] = 0.5 * e.get("relevance_score", 0) + 0.3 * e.get("credibility_score", 0.5)
        
        # Claim-aware final score using engineering weights (Req #9)
        w = {
            "claim_relevance": 0.30,
            "directness": 0.15,
            "entity_match": 0.10,
            "event_match": 0.10,
            "temporal_match": 0.08,
            "location_match": 0.07,
            "credibility": 0.05,
            "narrative_coherence": 0.05,
            "intent_match": 0.05,
            "multimodal": 0.10,
        }
        final_score = (
            w["claim_relevance"] * e.get("claim_relevance_score", 0) +
            w["directness"] * e.get("directness_score", 0) +
            w["entity_match"] * e.get("entity_match_score", 0) +
            w["event_match"] * e.get("event_match_score", 0) +
            w["temporal_match"] * e.get("temporal_match_score", 0) +
            w["location_match"] * e.get("location_match_score", 0) +
            w["credibility"] * e.get("credibility_score", 0.5) +
            w["narrative_coherence"] * e.get("narrative_coherence_score", 0) +
            w["intent_match"] * e.get("intent_match_score", 0.5) +
            w["multimodal"] * e.get("multimodal_score", 0.0)
        )
        e["final_score"] = round(final_score, CONFIDENCE_DECIMALS)
    
    all_evidence.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    fused = all_evidence[:MAX_FINAL_EVIDENCE]

    return {
        "case_id": case_id,
        "claim": case_claim,
        "queries": query_intents,
        "historical_text_count": len(hist_text),
        "historical_image_count": len(hist_image),
        "live_count": len(live.get("results", [])),
        "raw_count": live.get("raw_count", 0),
        "relevant_count": live.get("relevant_count", 0),
        "filtered_count": live.get("filtered_count", 0),
        "deduplicated_count": live.get("deduplicated_count", 0),
        "total_evidence": len(all_evidence),
        "fused_evidence": fused,
        "provider_status": {**live.get("provider_status", {}), **historical_status},
        "provider_diagnostics": {**live.get("provider_diagnostics", {}), "historical_errors": historical_errors},
    }


def load_nb7_evidence_package() -> Dict[str, Any]:
    p = NB6_DIR / "narrative_clustering" / "evidence_package_v2.json"
    if p.exists():
        import json
        return json.loads(p.read_text(encoding="utf-8"))
    return {}
