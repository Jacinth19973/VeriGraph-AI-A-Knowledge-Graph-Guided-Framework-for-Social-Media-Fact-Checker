
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re
import numpy as np
from sklearn.cluster import KMeans
from pipeline.embeddings import embed_text
DEFAULT_MAX_CLUSTERS = 4
DEFAULT_MIN_RELEVANCE = 0.45
DEFAULT_HIGH_RELEVANCE = 0.65
STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from",
    "were", "was", "are", "been", "have", "has", "had",
    "into", "about", "their", "there", "they", "them",
    "then", "than", "also", "more", "most", "some",
}
def _safe_float(value: Any, default: Optional[float] = None):
    try:
        if value is None:
            return default
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)

    norm = np.linalg.norm(vec)

    if norm <= 1e-12:
        return np.zeros_like(vec)

    return vec / norm
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom <= 1e-12:
        return 0.0

    return float(np.dot(a, b))

def _evidence_text(e: Dict[str, Any]) -> str:
    return (
        f"{e.get('title', '')}. "
        f"{e.get('snippet', '')}"
    ).strip()

def _extract_theme_keywords(evidence_items: List[Dict[str, Any]], cluster_indices: List[int], max_keywords: int = 8) -> List[str]:
    """Extract meaningful theme keywords from cluster evidence titles and snippets."""
    from collections import Counter

    stop_words = set(STOP_WORDS) | {
        "claim", "evidence", "news", "report", "photo", "image", "video",
        "says", "said", "according", "click", "watch", "read", "view",
        "online", "com", "www", "http", "https", "page", "website",
        "update", "updates", "share", "post", "posted", "article"
    }

    word_counts: Counter = Counter()
    for idx in cluster_indices:
        e = evidence_items[idx]
        text = f"{e.get('title', '')} {e.get('snippet', '')}"
        for word in re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()):
            if word not in stop_words:
                word_counts[word] += 1

    return [w for w, _ in word_counts.most_common(max_keywords)]

ANCHOR_STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from",
    "were", "was", "are", "been", "have", "has", "had",
    "into", "about", "their", "there", "they", "them",
    "then", "than", "also", "more", "most", "some",
    "under", "over", "after", "before", "during", "while",
    "new", "one", "two", "three", "four", "five",
    "first", "second", "third", "last", "next",
    "building", "built", "theme", "themes",
    "held", "hosted", "hosts", "host", "held",
    "resilience", "innovation", "cooperation", "sustainability",
    "september", "october", "november", "december",
    "january", "february", "march", "april", "may", "june",
    "july", "august",
}

_LOCATION_VOCAB = [
    "New Delhi", "New York", "United States", "United Kingdom",
    "South Africa", "Saudi Arabia", "South Korea", "North Korea",
    "Costa Rica", "Sri Lanka", "West Bengal", "Telangana",
    "Andhra Pradesh", "Uttar Pradesh", "Madhya Pradesh",
    "Punjab", "Haryana", "Rajasthan", "Maharashtra", "Tamil Nadu",
    "Karnataka", "Kerala", "Gujarat", "Bihar", "Odisha", "Assam",
    "Goa", "Agra", "Jaipur", "Lucknow", "Bhopal", "Nagpur",
    "Delhi", "Mumbai", "Chennai", "Bangalore", "Hyderabad",
    "Kolkata", "Ahmedabad", "Pune", "Surat", "Indore", "Bhubaneswar",
    "London", "Paris", "Berlin", "Madrid", "Rome", "Moscow",
    "Washington", "Beijing", "Tokyo", "Seoul", "Ottawa",
    "Canberra", "Brasilia", "Mexico City", "Jakarta", "Bangkok",
    "Singapore", "Dubai", "Geneva", "Brussels", "Vienna", "Islamabad",
    "Dhaka", "Kathmandu", "Colombo", "Beijing",
    "India", "Bharat", "Pakistan", "Bangladesh", "Nepal", "Sri Lanka",
    "China", "Japan", "Russia", "Brazil", "Mexico", "Egypt",
    "Nigeria", "Kenya", "Australia", "Canada", "France", "Germany",
    "Italy", "Spain", "Sweden", "Norway", "Denmark", "Finland",
]

_EVENT_VOCAB = [
    "summit", "conference", "meeting", "election", "tournament",
    "launch", "agreement", "chairmanship", "bilateral",
    "multilateral", "forum", "expo", "congress", "assembly",
    "treaty", "accord", "festival", "ceremony", "deal",
    "negotiation", "talks", "visit", "tour", "match", "war",
]


def _extract_claim_anchors(claim: str) -> Dict[str, List[str]]:
    """Extract claim anchors used to validate evidence relevance.

    Locations are matched as phrases (longest first) so that multi-word
    names such as "New Delhi" never degrade into the single token "New".
    """
    text = claim or ""

    years = re.findall(r"\b(?:19|20)\d{2}\b", text)

    # Locations: scan vocabulary longest-first so multi-word names win.
    locations = []
    for loc in sorted(_LOCATION_VOCAB, key=len, reverse=True):
        if re.search(rf"\b{re.escape(loc)}\b", text, re.IGNORECASE):
            if loc not in locations:
                locations.append(loc)

    event_terms = [
        term for term in _EVENT_VOCAB
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE)
    ]

    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
    multiword = re.findall(r"\b(?:[A-Z][a-z]+ ){1,3}[A-Z][a-z]+\b", text)
    single = re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", text)

    entities = []
    for term in acronyms + multiword:
        if term not in entities:
            entities.append(term)
    for term in single:
        if term.lower() in ANCHOR_STOP_WORDS:
            continue
        if term in entities:
            continue
        # Drop single words already covered by a multi-word entity.
        if any(term in phrase.split() for phrase in multiword):
            continue
        entities.append(term)

    return {
        "years": list(dict.fromkeys(years)),
        "locations": locations,
        "event_terms": event_terms,
        "entities": entities,
    }


def _anchor_score(
    claim: str,
    evidence_text: str,
) -> float:

    anchors = _extract_claim_anchors(claim)

    evidence_lower = evidence_text.lower()

    scores = []

    for entity in anchors["entities"]:
        # Use word-boundary regex to avoid partial matches (e.g. "India" != "Indian")
        pattern = r"\b" + re.escape(entity.lower()) + r"\b"
        if re.search(pattern, evidence_lower):
            scores.append(1.0)

    for year in anchors["years"]:
        if re.search(r"\b" + re.escape(year) + r"\b", evidence_text):
            scores.append(1.0)

    for location in anchors["locations"]:
        # Multi-word locations need phrase matching
        pattern = r"\b" + re.escape(location.lower()) + r"\b"
        if re.search(pattern, evidence_lower):
            scores.append(1.0)

    for event in anchors["event_terms"]:
        pattern = r"\b" + re.escape(event) + r"\b"
        if re.search(pattern, evidence_lower):
            scores.append(1.0)

    if not scores:
        return 0.0

    return float(np.mean(scores))


def _specific_anchor_ok(claim: str, evidence_text: str) -> bool:
    """True when the evidence mentions something specific to the claim.

    A generic event word ("summit") or a place ("New Delhi") alone also appears in
    unrelated stories (e.g. an AI summit in the same city), so it cannot qualify
    evidence. If the claim names a specific entity (BRICS), the evidence must
    mention it; otherwise at least two different anchor kinds must match.
    """
    anchors = _extract_claim_anchors(claim)
    generic = {l.lower() for l in anchors["locations"]} | {t.lower() for t in _EVENT_VOCAB}
    specific = [
        e for e in anchors["entities"]
        if e.lower() not in generic and e.lower() not in ANCHOR_STOP_WORDS
    ]
    text = (evidence_text or "").lower()

    def _has(term: str) -> bool:
        return re.search(r"\b" + re.escape(term.lower()) + r"\b", text) is not None

    if specific:
        return any(_has(e) for e in specific)

    kinds = 0
    kinds += any(_has(y) for y in anchors["years"])
    kinds += any(_has(l) for l in anchors["locations"])
    kinds += any(_has(t) for t in anchors["event_terms"])
    return kinds >= 2


def _claim_relevance(
    claim_embedding: np.ndarray,
    evidence_embedding: np.ndarray,
) -> float:
    """Semantic claim relevance in [0, 1].

    Both vectors are L2-normalised, so cosine similarity already lies in
    [-1, 1] and is ~[0, 1] for same-language text. It is used directly
    instead of remapped via (score + 1) / 2: that remap compresses the whole
    usable range into roughly [0.73, 0.97] for BGE and destroys the
    discrimination between relevant and irrelevant evidence.
    """
    score = _cosine_similarity(
        claim_embedding,
        evidence_embedding,
    )

    return float(np.clip(score, 0.0, 1.0))


def _should_retain_evidence(
    semantic_score: float,
    anchor_score: float,
    min_relevance: float = DEFAULT_MIN_RELEVANCE,
    high_relevance: float = DEFAULT_HIGH_RELEVANCE,
    specific_anchor_ok: bool = True,
) -> bool:
    """Two-stage relevance gate: semantic similarity AND claim anchors.

    Anchor evidence is mandatory. A high BGE cosine alone is not sufficient
    because sentence-transformer embeddings share a large common component,
    so topically unrelated English text still scores well above 0.6. Requiring
    at least one claim anchor (entity / year / location / event term) is what
    actually separates claim-relevant evidence from unrelated retrieval noise.
    """
    if anchor_score <= 0.0 or not specific_anchor_ok:
        return False

    if semantic_score >= high_relevance:
        return True

    if semantic_score >= min_relevance:
        return True

    return False


def _select_k(n: int) -> int:

    if n <= 1:
        return 1

    if n == 2:
        return 1

    return min(DEFAULT_MAX_CLUSTERS, n)


def _cluster_embeddings(
    embeddings: np.ndarray,
    k: int,
) -> np.ndarray:

    if len(embeddings) <= 1:
        return np.zeros(len(embeddings), dtype=int)

    if k <= 1:
        return np.zeros(len(embeddings), dtype=int)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300,
    )

    return _merge_similar_clusters(embeddings, model.fit_predict(embeddings))


def _merge_similar_clusters(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Merge clusters whose centroids are almost the same story (cosine similarity >= threshold).

    With only a handful of evidence items KMeans puts every item in its own cluster; articles that
    all report the same event should form one narrative.
    """
    threshold = float(__import__("os").getenv("NARRATIVE_MERGE_SIMILARITY", "0.80"))
    ids = sorted(set(int(x) for x in labels))
    if len(ids) <= 1:
        return labels
    cents = {}
    for cid in ids:
        c = embeddings[labels == cid].mean(axis=0)
        n = np.linalg.norm(c)
        cents[cid] = c / n if n else c
    parent = {cid: cid for cid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if float(np.dot(cents[a], cents[b])) >= threshold:
                parent[find(b)] = find(a)
    remap = {root: n for n, root in enumerate(sorted({find(c) for c in ids}))}
    return np.array([remap[find(int(x))] for x in labels], dtype=int)


def _representative_index(
    cluster_indices: List[int],
    embeddings: np.ndarray,
) -> int:

    cluster_matrix = embeddings[cluster_indices]

    centroid = np.mean(
        cluster_matrix,
        axis=0,
    )

    centroid = _normalize(centroid)

    distances = []

    for idx in cluster_indices:
        similarity = _cosine_similarity(
            embeddings[idx],
            centroid,
        )

        distance = 1.0 - similarity

        distances.append(distance)

    best_position = int(np.argmin(distances))

    return cluster_indices[best_position]


def _build_cluster(
    cluster_id: int,
    cluster_indices: List[int],
    evidence_items: List[Dict[str, Any]],
    embeddings: np.ndarray,
) -> Dict[str, Any]:

    representative_idx = _representative_index(
        cluster_indices,
        embeddings,
    )

    representative = evidence_items[representative_idx]

    relevance_scores = [
        _safe_float(
            evidence_items[i].get(
                "claim_relevance_score",
            ),
            0.0,
        )
        for i in cluster_indices
    ]

    credibility_scores = [
        _safe_float(
            evidence_items[i].get(
                "credibility_score",
            ),
            0.0,
        )
        for i in cluster_indices
    ]

    coherence_scores = []

    cluster_matrix = embeddings[cluster_indices]

    centroid = _normalize(
        np.mean(cluster_matrix, axis=0)
    )

    for idx in cluster_indices:
        coherence_scores.append(
            _cosine_similarity(
                embeddings[idx],
                centroid,
            )
        )

    domains = sorted(
        {
            e.get("domain")
            for e in (
                evidence_items[i]
                for i in cluster_indices
            )
            if e.get("domain")
        }
    )

    evidence_ids = [
        evidence_items[i].get(
            "evidence_id",
            "",
        )
        for i in cluster_indices
    ]

    return {
        "cluster_id": int(cluster_id),
        "size": int(len(cluster_indices)),

        "representative_evidence_id":
            str(representative.get(
                "evidence_id",
                "",
            )),

        "representative_title":
            str(representative.get(
                "title",
                "",
            )),

        "representative_text":
            _evidence_text(representative),

        "representative_relevance":
            _safe_float(
                representative.get(
                    "claim_relevance_score",
                ),
                0.0,
            ),

        "mean_relevance":
            float(np.mean(relevance_scores))
            if relevance_scores else 0.0,

        "mean_cluster_coherence":
            float(np.mean(coherence_scores))
            if coherence_scores else 0.0,

        "mean_credibility":
            float(np.mean(credibility_scores))
            if credibility_scores else 0.0,

        "source_count":
            int(len(domains)),

        "unique_source_domains":
            [str(d) for d in domains],

        "evidence_ids":
            [str(eid) for eid in evidence_ids],

        "evidence_sufficiency":
            (
                "high"
                if len(cluster_indices) >= 3
                else "medium"
                if len(cluster_indices) == 2
                else "low"
            ),

        "theme_keywords":
            _extract_theme_keywords(evidence_items, cluster_indices),

        "theme":
            ", ".join(
                _extract_theme_keywords(evidence_items, cluster_indices)[:5]
            ) or "No theme",

        "representative_evidence": [
            {
                "evidence_id": str(evidence_items[i].get("evidence_id", "")),
                "title": str(evidence_items[i].get("title", "")),
                "fusion_score": float(evidence_items[i].get("fusion_score", 0.0) or 0.0),
            }
            for i in [representative_idx]
        ],

        "status": "ACCEPTED",
    }


def cluster_narratives(
    evidence_items: List[Dict[str, Any]],
    case_claim: str = "",
) -> Dict[str, Any]:

    """
    Claim-conditioned narrative clustering following CRAVE principles.
    
    Pipeline:
    1. Claim embedding + Evidence embedding
    2. Claim relevance scoring (cosine similarity)
    3. Two-stage relevance gate (semantic + anchor check)
    4. K-means clustering on retained evidence (K=min(4, n))
    5. Centroid-nearest representative evidence selection
    6. Narrative acceptance filtering
    7. Grounded narrative generation from representative evidence
    
    Returns clusters, grounded narratives, and diagnostics.
    """
    
    if not evidence_items:
        return {
            "clusters": [],
            "evidence_with_clusters": [],
            "grounded_narratives": [],
            "diagnostics": {
                "status": "NO_EVIDENCE",
                "input_evidence": 0,
                "relevant_evidence": 0,
                "filtered_evidence": 0,
            },
        }

    # ============ STEP A: CLAIM EMBEDDING ============
    claim_embedding = np.asarray(
        embed_text(case_claim),
        dtype=np.float32,
    )
    claim_embedding = _normalize(claim_embedding)

    # ============ STEP B: EVIDENCE EMBEDDINGS ============
    relevant_items = []
    filtered_items = []
    evidence_embeddings = []

    for evidence in evidence_items:

        text = _evidence_text(evidence)

        if not text:
            evidence["narrative_status"] = "REJECTED_EMPTY"
            filtered_items.append(evidence)
            continue

        embedding = np.asarray(
            embed_text(text),
            dtype=np.float32,
        )

        embedding = _normalize(embedding)

        # ============ STEP C: CLAIM RELEVANCE SCORE ============
        semantic_score = _claim_relevance(
            claim_embedding,
            embedding,
        )

        # ============ STEP D: 5W1H CLAIM CONSISTENCY ============
        anchor_score = _anchor_score(
            case_claim,
            text,
        )

        specific_ok = _specific_anchor_ok(case_claim, text)
        evidence["specific_anchor_match"] = bool(specific_ok)

        # Store scores in evidence for downstream use
        evidence["claim_relevance_score"] = round(semantic_score, 4)
        evidence["claim_anchor_score"] = round(anchor_score, 4)

        # ============ STEP E: CLAIM RELEVANCE GATE ============
        if _should_retain_evidence(
            semantic_score,
            anchor_score,
            specific_anchor_ok=specific_ok,
        ):
            evidence["narrative_status"] = "RETAINED"

            relevant_items.append(evidence)
            evidence_embeddings.append(embedding)

        else:
            evidence["narrative_status"] = "REJECTED_LOW_RELEVANCE"

            filtered_items.append(evidence)

    # ============ Check if we have any relevant evidence ============
    if not relevant_items:

        return {
            "clusters": [],
            "evidence_with_clusters": relevant_items + filtered_items,
            "grounded_narratives": [],
            "diagnostics": {
                "status": "NO_RELEVANT_EVIDENCE",
                "input_evidence": len(evidence_items),
                "relevant_evidence": 0,
                "filtered_evidence": len(filtered_items),
            },
        }

    # ============ STEP F: SELECT K ============
    embeddings = np.asarray(evidence_embeddings, dtype=np.float32)
    k = _select_k(len(relevant_items))

    # ============ STEP G: K-MEANS CLUSTERING ============
    labels = _cluster_embeddings(embeddings, k)

    # Assign cluster_id to each retained evidence item
    for i, evidence in enumerate(relevant_items):
        evidence["cluster_id"] = int(labels[i])

    # ============ STEP H: BUILD CLUSTERS ============
    clusters = []

    for cluster_id in sorted({int(x) for x in labels}):

        indices = [
            i
            for i, label
            in enumerate(labels)
            if label == cluster_id
        ]

        cluster = _build_cluster(
            cluster_id,
            indices,
            relevant_items,
            embeddings,
        )

        # Narrative-level acceptance gate
        if (
            cluster["representative_relevance"] < DEFAULT_MIN_RELEVANCE
            or cluster["mean_relevance"] < DEFAULT_MIN_RELEVANCE
        ):
            cluster["status"] = "REJECTED_LOW_RELEVANCE"

        clusters.append(cluster)

    # ============ STEP I: SELECT ACCEPTED CLUSTERS ============
    accepted_clusters = [
        c
        for c in clusters
        if c["status"] == "ACCEPTED"
    ]

    # ============ STEP J: GENERATE GROUNDED NARRATIVES ============
    grounded_narratives = []

    for cluster in accepted_clusters:
        rep_title = cluster.get("representative_title", "")
        rep_text = cluster.get("representative_text", "")
        rep_id = cluster.get("representative_evidence_id", "")
        evidence_ids = cluster.get("evidence_ids", [])
        
        # Clean snippet from web noise
        clean_text = re.sub(r"https?://\S+", "", rep_text)
        clean_text = re.sub(r"\b(click here|watch video|read more|subscribe|all rights reserved)\b", "", clean_text, flags=re.I)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        
        # Construct meaningful grounded narrative
        if rep_title and clean_text and not clean_text.lower().startswith(rep_title.lower()):
            synthesized_narrative = f"{rep_title}. {clean_text}"
        else:
            synthesized_narrative = clean_text or rep_title or f"Verified narrative on {cluster.get('theme', 'claim topic')}."

        grounded_narratives.append(
            {
                "cluster_id": cluster["cluster_id"],

                "narrative_text": synthesized_narrative,

                "representative_evidence_id": rep_id,

                "evidence_ids": evidence_ids,

                "claim_relevance_score": cluster["representative_relevance"],

                "coherence_score": cluster["mean_cluster_coherence"],

                "mean_relevance": cluster["mean_relevance"],

                "mean_cluster_coherence": cluster["mean_cluster_coherence"],

                "stance_polarity": cluster.get("stance_polarity", 0.0),

                "theme": cluster.get("theme", ""),

                "status": "ACCEPTED",
            }
        )

    # ============ STEP K: DIAGNOSTICS ============
    scored_evidence = relevant_items + filtered_items

    global_coherence = (
        float(np.mean([c["mean_cluster_coherence"] for c in clusters]))
        if clusters else 0.0
    )

    global_credibility = (
        float(np.mean([c["mean_credibility"] for c in clusters]))
        if clusters else 0.0
    )

    overall_sufficiency = (
        "high" if len(relevant_items) >= 5
        else "medium" if len(relevant_items) >= 2
        else "low"
    )

    return {
        "clusters": clusters,

        "evidence_with_clusters": scored_evidence,

        "fused_evidence": scored_evidence,

        "grounded_narratives": grounded_narratives,

        "overall_evidence_sufficiency": overall_sufficiency,

        "global_cluster_coherence": global_coherence,

        "global_credibility": global_credibility,

        "diagnostics": {
            "status": "SUCCESS",

            "input_evidence": len(evidence_items),

            "relevant_evidence": len(relevant_items),

            "filtered_evidence": len(filtered_items),

            "total_evidence": len(scored_evidence),

            "cluster_count": len(clusters),

            "accepted_narratives": len(grounded_narratives),

            "rejected_narratives": len(clusters) - len(grounded_narratives),
        },
    }