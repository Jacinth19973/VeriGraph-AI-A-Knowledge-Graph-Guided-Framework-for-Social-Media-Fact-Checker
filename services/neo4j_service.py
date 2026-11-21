"""VeriGraph AI — Neo4j service with bounded KG retrieval."""
from __future__ import annotations
import csv
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from config.settings import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    KG_DIR,
)

logger = logging.getLogger(__name__)

_driver = None
_kg_metadata: Optional[Dict[str, Dict]] = None
_kg_metadata_loaded = False

# Hard limits (requirements)
MAX_ENTITY_MATCHES = 5
MAX_RELATIONSHIPS = 30
MAX_TOTAL_NODES = 30
MAX_QUERY_TIME = 15.0  # seconds

KG_METADATA_PATH = KG_DIR / "kg_case_metadata.csv"


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=300,
            max_connection_pool_size=10,
            connection_timeout=10.0,
        )
    return _driver


def _load_kg_metadata() -> Dict[str, Dict]:
    """Load kg_case_metadata.csv into memory for instant dataset case lookup."""
    global _kg_metadata, _kg_metadata_loaded
    if _kg_metadata_loaded:
        return _kg_metadata or {}

    _kg_metadata_loaded = True
    _kg_metadata = {}

    try:
        with open(KG_METADATA_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row.get("sample_id", "").strip()
                if not sample_id:
                    continue

                def parse_json_list(s: str) -> List[str]:
                    s = (s or "").strip()
                    if not s or s == "[]":
                        return []
                    try:
                        return json.loads(s)
                    except Exception:
                        return [x.strip().strip('"') for x in s.strip("[]").split(",") if x.strip()]

                _kg_metadata[sample_id] = {
                    "entities": parse_json_list(row.get("entities", "[]")),
                    "persons": parse_json_list(row.get("persons", "[]")),
                    "organizations": parse_json_list(row.get("organizations", "[]")),
                    "locations": parse_json_list(row.get("locations", "[]")),
                    "events": parse_json_list(row.get("events", "[]")),
                    "dates": parse_json_list(row.get("temporal_expression", "[]")),
                    "sources": [row.get("source", "").strip()] if row.get("source", "").strip() else [],
                    "source_domains": [row.get("source_domain", "").strip()] if row.get("source_domain", "").strip() else [],
                    "relationships": [],
                    "related_claims": [],
                }
        logger.info(f"[KG] Loaded metadata for {len(_kg_metadata)} cases from CSV")
    except Exception as e:
        logger.warning(f"[KG] Failed to load kg_case_metadata.csv: {e}")
        _kg_metadata = {}

    return _kg_metadata


def _log(msg: str):
    """Log with [KG] prefix for pipeline visibility."""
    logger.info(f"[KG] {msg}")


def check_neo4j_health() -> bool:
    """Quick connectivity test with logging."""
    try:
        driver = get_driver()
        driver.verify_connectivity()
        _log("Neo4j connectivity OK")
        return True
    except Exception as e:
        _log(f"Neo4j connectivity FAILED: {e}")
        return False


def _extract_candidate_entities(claim: str, evidence_items: List[Dict] = None) -> List[str]:
    """Extract candidate entity names from claim and evidence snippets."""
    import re
    candidates: List[str] = []

    def _add(term: str) -> None:
        if term not in candidates:
            candidates.append(term)

    # Acronyms first (BRICS, NATO): the plain capitalised-word pattern below misses them
    for acronym in re.findall(r'\b[A-Z]{2,}\b', claim):
        _add(acronym)

    # From claim
    for word in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim):
        if len(word) > 2 and word.lower() not in {"the", "and", "for", "with", "this", "that", "shows", "image", "photo", "video", "claim", "claims"}:
            _add(word)

    # From evidence snippets/titles (top 3)
    if evidence_items:
        for e in evidence_items[:3]:
            text = f"{e.get('title', '')} {e.get('snippet', '')}"
            for word in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
                if len(word) > 2:
                    _add(word)

    return candidates[:10]  # Cap candidates to search


def _query_entities_by_names(session, names: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Lookup entities by name (indexed), return 1-hop relationships.
    Returns (nodes, relationships) with hard limits.
    """
    if not names:
        return [], []

    nodes = []
    relationships = []
    seen_node_ids = set()

    # Parameterized query with LIMIT per name + overall LIMIT
    cypher = """
    MATCH (e) WHERE e.name IN $names
    OPTIONAL MATCH (e)-[r]-(connected)
    WHERE size((e)-[]-()) > 0
    RETURN e.node_id AS id, e.name AS name, labels(e) AS labels,
           collect(DISTINCT {type: type(r), target: connected.name, target_id: connected.node_id, target_labels: labels(connected)}) AS rels
    LIMIT $max_nodes
    """

    try:
        result = session.run(
            cypher,
            {"names": names[:MAX_ENTITY_MATCHES], "max_nodes": MAX_TOTAL_NODES},
            read_timeout=MAX_QUERY_TIME
        ).data()

        for row in result:
            nid = row["id"]
            if nid in seen_node_ids:
                continue
            seen_node_ids.add(nid)

            nodes.append({
                "node_id": nid,
                "name": row["name"],
                "labels": row["labels"],
            })

            for rel in row["rels"]:
                if rel.get("target") and len(relationships) < MAX_RELATIONSHIPS:
                    relationships.append({
                        "source": row["name"],
                        "type": rel["type"],
                        "target": rel["target"],
                    })

    except Neo4jError as e:
        _log(f"Entity lookup query error: {e}")

    return nodes[:MAX_TOTAL_NODES], relationships[:MAX_RELATIONSHIPS]


def _query_case_metadata(sample_id: str) -> Optional[Dict[str, Any]]:
    """Instant lookup from pre-computed CSV metadata."""
    metadata = _load_kg_metadata()
    return metadata.get(sample_id)


def query_kg_context(
    case_id: str = None,
    claim: str = None,
    evidence_items: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Bounded KG context retrieval.
    
    Strategy:
    1. If case_id matches a known dataset sample_id → return CSV metadata (instant)
    2. Else for new claims: extract candidate entities → bounded Neo4j lookup
    3. On any failure/timeout → return {"status": "unavailable", ...}
    """
    start_time = time.time()
    _log("START")

    # Initialize result structure
    result = {
        "status": "ok",
        "entities": [], "persons": [], "organizations": [],
        "locations": [], "events": [], "dates": [],
        "sources": [], "source_domains": [],
        "relationships": [], "related_claims": [],
        "nodes_inspected": 0,
        "nodes_returned": 0,
        "relationships_returned": 0,
        "duration_seconds": 0.0,
        "neo4j_used": False,
    }

    # 1. FAST PATH: Check CSV metadata for existing dataset cases
    # case_id from orchestrator is often the sample_id (e.g., "MMF_train_000000")
    if case_id:
        meta = _query_case_metadata(case_id)
        if meta:
            result.update(meta)
            result["nodes_returned"] = len(meta.get("entities", []))
            result["duration_seconds"] = round(time.time() - start_time, 3)
            _log(f"CSV metadata hit for {case_id}")
            _log(f"Nodes returned: {result['nodes_returned']}")
            _log(f"Duration: {result['duration_seconds']} seconds")
            _log("END")
            return result

    # 2. NEW CLAIM PATH: Bounded Neo4j retrieval
    _log("Entity extraction complete")
    
    # Check Neo4j connectivity before main query
    if not check_neo4j_health():
        result["status"] = "unavailable"
        result["reason"] = "Neo4j connection failed"
        result["duration_seconds"] = round(time.time() - start_time, 3)
        _log("Neo4j unavailable")
        _log("END")
        return result

    _log("Query started")
    
    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE, default_access_mode="READ") as session:
            # Extract candidate entities from claim + evidence
            candidates = _extract_candidate_entities(claim or "", evidence_items or [])
            _log(f"Candidate entities: {candidates}")

            if not candidates:
                result["status"] = "unavailable"
                result["reason"] = "No candidate entities extracted from claim"
                result["duration_seconds"] = round(time.time() - start_time, 3)
                _log("No candidates to query")
                _log("END")
                return result

            # Bounded lookup + 1-hop traversal
            nodes, relationships = _query_entities_by_names(session, candidates)
            
            result["neo4j_used"] = True
            result["nodes_inspected"] = len(nodes)
            
            # Categorize nodes by label
            for node in nodes:
                labels = node.get("labels", [])
                name = node.get("name", "")
                if not name:
                    continue
                if "Person" in labels:
                    result["persons"].append(name)
                elif "Organization" in labels:
                    result["organizations"].append(name)
                elif "Location" in labels:
                    result["locations"].append(name)
                elif "Event" in labels:
                    result["events"].append(name)
                elif "Time" in labels or "Date" in labels:
                    result["dates"].append(name)
                result["entities"].append(name)

            # Deduplicate
            for key in ["entities", "persons", "organizations", "locations", "events", "dates"]:
                result[key] = list(dict.fromkeys(result[key]))

            result["relationships"] = relationships
            result["nodes_returned"] = len(nodes)
            result["relationships_returned"] = len(relationships)

    except Neo4jError as e:
        _log(f"Neo4j query error: {e}")
        result["status"] = "unavailable"
        result["reason"] = f"Neo4j error: {e}"
    except Exception as e:
        _log(f"Unexpected error: {e}")
        result["status"] = "unavailable"
        result["reason"] = f"Unexpected error: {e}"

    result["duration_seconds"] = round(time.time() - start_time, 3)
    _log(f"Query completed")
    _log(f"Nodes returned: {result['nodes_returned']}")
    _log(f"Relationships returned: {result['relationships_returned']}")
    _log(f"Duration: {result['duration_seconds']} seconds")
    _log("END")

    return result


# Backward compatibility
def query_subgraph_for_case(case_id: str = None, claim: str = None) -> Dict[str, Any]:
    """Legacy function - delegates to new bounded retrieval."""
    return query_kg_context(case_id=case_id, claim=claim)


def add_candidate_nodes(candidates: List[Dict[str, Any]], case_id: str) -> int:
    driver = get_driver()
    added = 0
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for c in candidates:
                try:
                    query = (
                        "MERGE (n:Entity {name: $name}) "
                        "SET n += $props, n.created_by_case = $case_id, "
                        "n.validation_status = $status"
                    )
                    session.run(query, {
                        "name": c.get("name", ""),
                        "props": c.get("properties", {}),
                        "case_id": case_id,
                        "status": c.get("validation_status", "CANDIDATE"),
                    })
                    added += 1
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"add_candidate_nodes error: {e}")
    return added


def get_kg_stats() -> Dict[str, int]:
    driver = get_driver()
    stats = {"nodes": 0, "relationships": 0}
    try:
        with driver.session(database=NEO4J_DATABASE, default_access_mode="READ") as session:
            stats["nodes"] = session.run("MATCH (n) RETURN count(n)").single()[0]
            stats["relationships"] = session.run("MATCH ()-[r]->() RETURN count(r)").single()[0]
    except Exception as e:
        logger.warning(f"get_kg_stats error: {e}")
    return stats