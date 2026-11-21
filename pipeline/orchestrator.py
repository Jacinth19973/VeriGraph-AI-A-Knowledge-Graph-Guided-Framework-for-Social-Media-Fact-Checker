"""VeriGraph AI — Pipeline orchestrator (progress events, no notebooks)."""
from __future__ import annotations
from services.time_service import india_now, utc_iso_now, clear_overrides
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from services.json_utils import make_json_safe
from models.schemas import (
    AuditEntry, KGCandidate,
    NarrativeResult, RetrievalResult,
    SpecialistResultBundle, UncertaintyProfile,
    DebateResult, JudgeResult, ExplanationResult, KGExpansionResult, VerificationResult,
    SpecialistResult,
)
from storage.case_store import save_case


def _to_audit(entry: AuditEntry) -> AuditEntry:
    entry.end_time = utc_iso_now()
    if entry.start_time:
        try:
            s = datetime.fromisoformat(entry.start_time)
            e = datetime.fromisoformat(entry.end_time)
            entry.duration_seconds = round((e - s).total_seconds(), 2)
        except Exception:
            entry.duration_seconds = 0.0
    return entry


def _to_specialist(datum: Dict[str, Any]) -> SpecialistResult:
    return SpecialistResult(
        agent=datum.get("agent", ""),
        status=datum.get("status", "completed"),
        assessment=datum.get("assessment") or datum.get("overall_assessment", "insufficient"),
        confidence=float(datum.get("confidence", 0)),
        key_findings=datum.get("key_findings", []),
        uncertainty_flags=datum.get("uncertainty_flags", []),
        reasoning=datum.get("reasoning", ""),
        supporting_evidence_ids=datum.get("supporting_evidence_ids", []),
        refuting_evidence_ids=datum.get("refuting_evidence_ids", []),
        evidence_assessments=datum.get("evidence_assessments", []),
        source_assessments=datum.get("source_assessments", []),
        temporal_findings=datum.get("temporal_findings", []),
        location_findings=datum.get("location_findings", []),
        contradiction_evidence_ids=datum.get("contradiction_evidence_ids", []),
    )


def _bundle_specialists(spec: Dict[str, Any]) -> SpecialistResultBundle:
    return SpecialistResultBundle(
        evidence_verification=_to_specialist(spec.get("evidence_verification", {})),
        claim_verification=_to_specialist(spec.get("claim_verification", {})),
        source_credibility=_to_specialist(spec.get("source_credibility", {})),
        temporal_location_consistency=_to_specialist(spec.get("temporal_location_consistency", {})),
    )


def _spec_as_dict(bundle: SpecialistResultBundle) -> Dict[str, Dict[str, Any]]:
    return {name: getattr(bundle, name).dict() for name in
            ("evidence_verification", "claim_verification", "source_credibility", "temporal_location_consistency")}


# Ablation configurations used by the benchmark (the UI's Configs A-E). Normal runs use E (everything on).
CONFIG_FLAGS = {
    "A": {"narratives": False, "specialists": False, "debate": False, "kg": False, "consensus": False},  # retrieval + judge
    "B": {"narratives": True,  "specialists": False, "debate": False, "kg": False, "consensus": False},  # + narrative clusters
    "C": {"narratives": True,  "specialists": True,  "debate": False, "kg": False, "consensus": False},  # + specialists
    "D": {"narratives": True,  "specialists": True,  "debate": True,  "kg": False, "consensus": False},  # + debate
    "E": {"narratives": True,  "specialists": True,  "debate": True,  "kg": True,  "consensus": True},   # full VeriGraph
}


def verify_claim(case_input: Dict[str, Any],
                 progress_cb: Optional[Callable[[str, str], None]] = None) -> VerificationResult:
    """Run the full verification pipeline for a single case."""
    def report(stage: str, status: str):
        if progress_cb:
            progress_cb(stage, status)

    case = VerificationResult(
        case_id=case_input["case_id"],
        claim=case_input["claim"],
        image_path=case_input.get("image_path", ""),
        created_at=utc_iso_now(),
        input_metadata=case_input.get("metadata", {}),
    )
    cfg = str(case_input.get("config", "E")).upper()
    cfg = cfg if cfg in CONFIG_FLAGS else "E"
    flags = CONFIG_FLAGS[cfg]
    case.config_name = cfg
    case.pipeline_config = dict(flags)

    # === EMBEDDINGS ===
    report("embeddings", "start")
    t = AuditEntry(stage="embeddings", status="running")
    t.start_time = utc_iso_now()
    text_emb = None
    image_emb = None
    local_image = ""
    image_ref = case.image_path          # original reference (public URL, data URL or local path)
    try:
        from pipeline.embeddings import embed_text, embed_image
        text_emb = embed_text(case.claim)
        t.message = "text embedding computed"
        t.status = "completed"
        if case.image_path:
            try:
                if __import__("os").getenv("VERIGRAPH_USE_IMAGES", "true").strip().lower() in ("1", "true", "yes"):
                    # opt-in: download URL / decode upload to a local file so CLIP can read it
                    from services.image_fetch import ensure_local_image
                    local_image, note = ensure_local_image(case.image_path)
                    image_emb = embed_image(local_image) if local_image else None
                    t.message += f"; image {'embedded' if image_emb is not None else 'skipped'} ({note})"
                    if local_image and case.image_path.startswith("data:"):
                        case.image_path = local_image      # keep the saved record small
                else:
                    image_emb = embed_image(case.image_path)   # original behaviour: local paths only
            except Exception as img_exc:
                t.message += f"; image skipped ({type(img_exc).__name__}: {str(img_exc)[:120]})"
    except Exception as e:
        t.status = "failed"; t.error = str(e)
    case.audit.append(_to_audit(t))

    # === RETRIEVAL ===
    report("retrieval", "start")
    t = AuditEntry(stage="retrieval", status="running"); t.start_time = utc_iso_now()
    try:
        from pipeline.retrieval import build_evidence_package
        pkg = build_evidence_package(text_emb, image_emb, case.claim, case.case_id)
        case.retrieval = RetrievalResult(
            combined_evidence=pkg.get("fused_evidence", []),
            fused_evidence=pkg.get("fused_evidence", []),
            provider_status=pkg.get("provider_status", {}),
            evidence_count=pkg.get("total_evidence", 0),
        )
        t.message = f"{pkg.get('total_evidence', 0)} evidence items"
        t.status = "completed"
    except Exception as e:
        t.status = "failed"; t.error = str(e)
    case.audit.append(_to_audit(t))

    # === REVERSE IMAGE SEARCH ===
    report("reverse_image", "start")
    t = AuditEntry(stage="reverse_image", status="running"); t.start_time = utc_iso_now()
    reverse = {"status": "skipped", "results": []}
    try:
        if not image_ref or not flags["consensus"]:
            t.status, t.message = "skipped", ("no image supplied" if not image_ref else "disabled in this configuration")
        else:
            from services.reverse_image_service import reverse_image_search, reverse_results_to_evidence
            reverse = reverse_image_search(image_ref)
            added = reverse_results_to_evidence(reverse.get("results", []), case.case_id)
            for item, src in zip(added, reverse.get("results", [])):
                src["evidence_id"] = item["evidence_id"]
            case.retrieval.fused_evidence.extend(added)
            case.retrieval.evidence_count = len(case.retrieval.fused_evidence)
            case.retrieval.provider_status["serpapi_reverse_image"] = reverse["status"]
            t.status = "completed" if reverse["status"] in ("success", "empty") else (
                "skipped" if reverse["status"] in ("not_public", "unconfigured") else "degraded")
            t.message = (f"status={reverse['status']}, {reverse.get('result_count', 0)} pages show this image"
                         + (f" ({reverse['error']})" if reverse.get("error") else ""))
    except Exception as e:
        t.status = "failed"; t.error = str(e)
    case.audit.append(_to_audit(t))

    # === NARRATIVES ===
    report("narratives", "start")
    t = AuditEntry(stage="narratives", status="running"); t.start_time = utc_iso_now()
    try:
        if not (flags["narratives"]):
            t.status, t.message = "skipped", "disabled in this configuration"
        else:
            from pipeline.narrative import cluster_narratives
            narr = cluster_narratives(case.retrieval.fused_evidence, case_claim=case.claim)
            case.narratives = NarrativeResult(
                clusters=narr.get("clusters", []),
                grounded_narratives=narr.get("grounded_narratives", []),
                diagnostics=narr.get("diagnostics", {}),
                fused_evidence=narr.get("fused_evidence", []),
                overall_evidence_sufficiency=narr.get("overall_evidence_sufficiency", "unknown"),
                global_cluster_coherence=narr.get("global_cluster_coherence", 0.0),
                global_credibility=narr.get("global_credibility", 0.0),
            )
            t.status = "completed"
    except Exception as e:
        t.status = "failed"; t.error = str(e)
    case.audit.append(_to_audit(t))

    # Extract accepted narratives and evidence for downstream stages.
    # IMPORTANT: Do NOT discard all evidence just because no narrative was accepted.
    # Use relevance-filtered evidence for downstream stages; narrative clustering
    # organizes evidence but is not the sole gatekeeper for what reaches agents.
    accepted_narratives = case.narratives.grounded_narratives if case.narratives else []

    # Determine relevance-filtered evidence (items that passed the relevance gate)
    all_fused = case.retrieval.fused_evidence if case.retrieval else []
    filtered_fused_evidence = [
    e for e in all_fused
    if e.get("narrative_status") in ("RETAINED", None)
    ]
    # Fallback: if no relevance-filtered evidence, use all fused evidence
    if not filtered_fused_evidence and all_fused:
        filtered_fused_evidence = all_fused
    if not flags["narratives"]:                     # plain RAG baseline: just the top retrieved items, unfiltered
        filtered_fused_evidence = filtered_fused_evidence[:8]

    # Record which evidence IDs were retained for downstream processing
    accepted_evidence_ids = {e.get("evidence_id") for e in filtered_fused_evidence if e.get("evidence_id")}

    # === IMAGE CHECK (CLIP consistency + reverse-image context + visual filtering) ===
    report("image_check", "start")
    t = AuditEntry(stage="image_check", status="running"); t.start_time = utc_iso_now()
    try:
        if not image_ref or not flags["consensus"]:
            t.status, t.message = "skipped", ("no image supplied" if not image_ref else "disabled in this configuration")
        else:
            from pipeline.multimodal import analyze_image_context
            case.multimodal = analyze_image_context(case.claim, local_image, filtered_fused_evidence, all_fused, reverse)
            t.status = "completed"
            t.message = f"consistency={case.multimodal['consistency']}; " + "; ".join(case.multimodal["reasons"])[:300]
    except Exception as e:
        t.status = "failed"; t.error = str(e)
    case.audit.append(_to_audit(t))

    # === KG CONTEXT ===
    report("kg_context", "start")
    t = AuditEntry(stage="kg_context", status="running"); t.start_time = utc_iso_now()
    kg_context: Dict[str, Any] = {}
    try:
        if not (flags["kg"]):
            t.status, t.message = "skipped", "disabled in this configuration"
            kg_context = {"status": "skipped", "reason": "disabled in this configuration"}
        else:
            from services.neo4j_service import query_kg_context
            evidence_items = filtered_fused_evidence
            kg_context = query_kg_context(case_id=case.case_id, claim=case.claim, evidence_items=evidence_items)
            t.message = (f"{len(kg_context.get('entities', []))} entities, "
            f"{len(kg_context.get('events', []))} events, "
            f"status={kg_context.get('status')}, "
            f"neo4j={kg_context.get('neo4j_used')}, "
            f"duration={kg_context.get('duration_seconds')}s")
            if kg_context.get("status") == "unavailable":
                t.status = "degraded"
            else:
                t.status = "completed"
    except Exception as e:
        t.status = "degraded"; t.error = str(e)
    case.audit.append(_to_audit(t))

    case.kg_context = make_json_safe(kg_context) if kg_context else {"status": "unavailable", "reason": "KG context stage did not return data"}

    if not filtered_fused_evidence:
        return _finish_without_evidence(case, report)

    # === SPECIALISTS (SUPPORTER + SKEPTIC) ===
    report("specialists", "start")
    t = AuditEntry(stage="specialists", status="running"); t.start_time = utc_iso_now()
    try:
        if not (flags["specialists"]):
            t.status, t.message = "skipped", "disabled in this configuration"
        else:
            from pipeline.specialists import run_specialists
            # Pass only accepted evidence, not all retrieved evidence
            spec = run_specialists(filtered_fused_evidence, case.claim, case.case_id)
            case.specialists = _bundle_specialists(spec)
            _apply_stance_to_clusters(case)
            t.message = "all specialists completed"
            t.status = "completed"
    except Exception as e:
        t.status = "failed"; t.error = str(e)
    case.audit.append(_to_audit(t))

    # === UNCERTAINTY ===
    report("uncertainty", "start")
    t = AuditEntry(stage="uncertainty", status="running"); t.start_time = utc_iso_now()
    try:
        if not (flags["specialists"]):
            t.status, t.message = "skipped", "disabled in this configuration"
        else:
            from pipeline.uncertainty import compute_uncertainty
            # Use accepted narratives for uncertainty calculation
            narr_diag = {
                "support_signal_count": sum(c.support_signal_count for c in case.narratives.clusters) if case.narratives and case.narratives.clusters else 0,
                "conflict_signal_count": sum(c.conflict_signal_count for c in case.narratives.clusters) if case.narratives and case.narratives.clusters else 0,
                "overall_evidence_sufficiency": case.narratives.overall_evidence_sufficiency if case.narratives else "medium",
            }
            u = compute_uncertainty(_spec_as_dict(case.specialists),
                                    filtered_fused_evidence, narr_diag)
            case.uncertainty = UncertaintyProfile(**u)
            t.status = "completed"
    except Exception as e:
        t.status = "failed"; t.error = str(e)
    case.audit.append(_to_audit(t))

    # === DEBATE (SUPPORTER vs SKEPTIC) ===
    history: list = []
    runs = {"max_rounds": 2, "actual_rounds": 0, "stop_reason": "max_rounds_reached", "turns": []}
    report("debate", "start")
    t = AuditEntry(stage="debate", status="running"); t.start_time = utc_iso_now()
    try:
        if not (flags["debate"]):
            t.status, t.message = "skipped", "disabled in this configuration"
        else:
            from pipeline.debate import run_debate
            d = run_debate(filtered_fused_evidence, _spec_as_dict(case.specialists),
                           kg_context, case.uncertainty.dict(), case.claim)
            runs.update({k: v for k, v in d.items() if k != "history"})
            history = d.get("history", [])
            case.debate = DebateResult(**{k: v for k, v in d.items() if k != "history"})
            t.status = "completed"
    except Exception as e:
        t.status = "failed"; t.error = str(e)
    case.audit.append(_to_audit(t))

    # === JUDGE ===
    report("judge", "start")
    t = AuditEntry(stage="judge", status="running"); t.start_time = utc_iso_now()
    try:
        from pipeline.debate import run_judge
        j = run_judge(filtered_fused_evidence,
                      _spec_as_dict(case.specialists),
                      history,
                      kg_context,
                      case.uncertainty.dict(),
                      case.claim,
                      consensus=flags["consensus"])
        case.judge = JudgeResult(
            final_verdict=j.get("final_verdict", "Insufficient"),
            confidence=j.get("confidence", 0.0),
            supporter_confidence=j.get("supporter_final_confidence", 0.0),
            skeptic_confidence=j.get("skeptic_final_confidence", 0.0),
            reasoning=j.get("reasoning", ""),
            key_evidence_cited=j.get("key_evidence_cited", []),
            key_kg_gaps_cited=j.get("key_kg_gaps_cited", []),
            dissent_notes=j.get("dissent_notes", ""),
            label_reasoning_alignment=j.get("label_reasoning_alignment", {}),
            judge_status=j.get("judge_status", "success"),
            judge_error=j.get("judge_error", ""),
            temporal_analysis=j.get("temporal_analysis", {}),
            current_date=j.get("current_date", ""),
            current_date_ist=j.get("current_date_ist", ""),
        )
        t.status = "completed"
        t.message = f"judge_status={j.get('judge_status', 'success')}, verdict={j.get('final_verdict', '')}"
    except Exception as e:
        t.status = "failed"; t.error = str(e)
        case.judge = JudgeResult(final_verdict="Insufficient", confidence=0.0,
                                 judge_status="error", judge_error=str(e))
    case.audit.append(_to_audit(t))

    try:
        from pipeline.multimodal import apply_image_verdict
        if flags["consensus"]:
            apply_image_verdict(case)
    except Exception as e:
        print(f"[orchestrator] image verdict adjustment failed: {e}")

    # === EXPLAINABILITY ===
    report("explainability", "start")
    t = AuditEntry(stage="explainability", status="running"); t.start_time = utc_iso_now()
    try:
        from pipeline.explainability import build_explanations
        case.explanation = ExplanationResult(**build_explanations(
            _spec_as_dict(case.specialists), case.uncertainty.dict(), case.judge.dict(),
            filtered_fused_evidence, kg_context, runs["turns"], claim=case.claim))
        from pipeline.multimodal import apply_image_to_explanation
        apply_image_to_explanation(case)
        t.status = "completed"
    except Exception as e:
        t.status = "failed"; t.error = str(e)
    case.audit.append(_to_audit(t))

    # === KG EXPANSION ===
    report("kg_expansion", "start")
    t = AuditEntry(stage="kg_expansion", status="running"); t.start_time = utc_iso_now()
    try:
        if not (flags["kg"]):
            t.status, t.message = "skipped", "disabled in this configuration"
        else:
            from pipeline.kg_expansion import run_kg_expansion
            kg = run_kg_expansion(
                claim=case.claim,
                narratives=accepted_narratives,
                evidence_items=filtered_fused_evidence,
                verifier_result=case.judge.model_dump(),
                case_id=case.case_id,
                kg_context=kg_context,
            )
            case.kg_update = KGExpansionResult(
                candidates_generated=kg.get("candidates_generated", 0),
                candidates_validated=kg.get("candidates_validated", 0),
                candidates_rejected=kg.get("candidates_rejected", 0),
                candidates_promoted=kg.get("candidates_promoted", 0),
                candidates=[KGCandidate(**c) for c in kg.get("candidates", [])],
            )
            t.status = "completed" if kg.get("status") == "completed" else kg.get("status", "failed")
            t.message = (f"llm_status={kg.get('llm_status', 'n/a')}, generated={kg.get('candidates_generated', 0)}, "
                         f"validated={kg.get('candidates_validated', 0)}, rejected={kg.get('candidates_rejected', 0)}"
                         + (f", reason={kg.get('reason')}" if kg.get("reason") else ""))
    except Exception as e:
        t.status = "failed"; t.error = str(e)
    case.audit.append(_to_audit(t))

    return _persist(case, report)


def _apply_stance_to_clusters(case: VerificationResult) -> None:
    '''Fill each cluster's support/conflict counts from the stances the specialists produced.'''
    stance = {a.get("evidence_id"): str(a.get("stance", "")).lower()
              for a in (case.specialists.evidence_verification.evidence_assessments or [])}
    polarity = {}
    for c in case.narratives.clusters:
        c.support_signal_count = sum(1 for eid in c.evidence_ids if stance.get(eid) == "supports")
        c.conflict_signal_count = sum(1 for eid in c.evidence_ids if stance.get(eid) == "refutes")
        total = c.support_signal_count + c.conflict_signal_count
        polarity[c.cluster_id] = round((c.support_signal_count - c.conflict_signal_count) / total, 2) if total else 0
    for n in case.narratives.grounded_narratives:
        n["stance_polarity"] = polarity.get(n.get("cluster_id"), 0)


def _skipped_audit(stage: str, why: str) -> AuditEntry:
    now = utc_iso_now()
    return AuditEntry(stage=stage, status="skipped", start_time=now, end_time=now,
                      duration_seconds=0.0, message=why)


def _finish_without_evidence(case: VerificationResult, report) -> VerificationResult:
    """No usable evidence: skip the LLM stages instead of letting agents argue from nothing."""
    why = "skipped: no evidence was retrieved"
    skipped = lambda name: SpecialistResult(agent=name, status="skipped_no_evidence",
                                            assessment="insufficient", confidence=0.0,
                                            key_findings=["no evidence available to assess"])
    case.specialists = SpecialistResultBundle(
        evidence_verification=skipped("evidence_verification"),
        claim_verification=skipped("claim_verification"),
        source_credibility=skipped("source_credibility"),
        temporal_location_consistency=skipped("temporal_location_consistency"),
    )
    case.uncertainty = UncertaintyProfile(evidence_insufficiency=1.0, debate_priority="skipped")
    case.debate = DebateResult(max_rounds=0, actual_rounds=0, stop_reason="no_evidence", turns=[])
    case.judge = JudgeResult(
        final_verdict="Insufficient", confidence=0.0, judge_status="skipped_no_evidence",
        reasoning="No evidence was retrieved, so no verdict was attempted. Check the embeddings, "
                  "retrieval and provider status entries in the audit trail.",
    )
    case.explanation = ExplanationResult(
        summary="No evidence was retrieved for this claim, so no verdict could be reached.",
        verdict_reason=case.judge.reasoning,
        knowledge_gaps=["No evidence retrieved"],
        natural_language_summary="No evidence was retrieved for this claim, so no verdict could be reached.",
        confidence_definition="Verification Confidence",
    )
    for stage in ("specialists", "uncertainty", "debate", "judge", "explainability", "kg_expansion"):
        case.audit.append(_skipped_audit(stage, why))
    return _persist(case, report)


def _persist(case: VerificationResult, report) -> VerificationResult:
    report("persist", "start")
    # The evidence list is stored once, in retrieval.fused_evidence; drop the duplicate copies.
    case.retrieval.combined_evidence = []
    case.narratives.fused_evidence = []
    save_path = ""
    try:
        save_path = save_case(case)
    except Exception as e:
        print(f"[orchestrator] first save failed: {e}")
    try:
        case.processing_metrics = {
            "storage_path": str(save_path),
            "duration_seconds": _case_duration(case),
            "pipeline_version": "1.0",
            "embedding_model": "BGE-base-en-v1.5 + CLIP",
            "llm_model": __import__("os").getenv("OLLAMA_MODEL", "qwen3:4b"),
            "retrieval_providers": list(case.retrieval.provider_status.keys()) if case.retrieval else [],
            "neo4j_status": str((case.kg_context or {}).get("status", "unavailable")),
            "faiss_status": "unknown" if not case.retrieval else ("available" if case.retrieval.evidence_count else "no_results"),
            "ollama_status": case.judge.judge_status if case.judge else "unknown",
            "total_evidence_count": case.retrieval.evidence_count if case.retrieval else 0,
            "narrative_clusters_count": len(case.narratives.grounded_narratives) if case.narratives else 0,
            "specialist_calls": 0 if (case.judge and case.judge.judge_status == "skipped_no_evidence") else 4,
            "debate_rounds": case.debate.actual_rounds if case.debate else 0,
            "kg_expansion_candidates": case.kg_update.candidates_generated if case.kg_update else 0,
            "verdict": case.judge.final_verdict if case.judge else "INSUFFICIENT",
            "confidence": case.judge.confidence if case.judge else 0.0,
        }
        case = VerificationResult.model_validate(make_json_safe(case.model_dump(mode="python")))
        save_path = save_case(case)
    except Exception as e:
        print(f"[orchestrator] persist failed: {e}")
    report("persist", "completed")
    return VerificationResult.model_validate(make_json_safe(case.model_dump(mode="python")))


def _case_duration(case: VerificationResult) -> float:
    total = sum(e.duration_seconds for e in case.audit if e.duration_seconds)
    return round(total, 2)
